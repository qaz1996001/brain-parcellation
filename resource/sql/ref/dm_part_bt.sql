CREATE DATABASE  IF NOT EXISTS `mes_omi` /*!40100 DEFAULT CHARACTER SET utf8 */;
USE `mes_omi`;
-- MySQL dump 10.13  Distrib 5.7.12, for Win32 (AMD64)
--
-- Host: 127.0.0.1    Database: mes_omi
-- ------------------------------------------------------
-- Server version	5.5.5-10.2.7-MariaDB

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `dm_part_bt`
--

DROP TABLE IF EXISTS `dm_part_bt`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `dm_part_bt` (
  `part_id` varchar(24) NOT NULL,
  `part_raw` varchar(16) DEFAULT NULL,
  `route_id` varchar(24) NOT NULL DEFAULT 'ROUTE-ZZZ-01' COMMENT 'for user not need to create each part / step record -- 2020/03/26 lkchena',
  `rd_flag` int(11) NOT NULL DEFAULT 0 COMMENT '2020/04/14 lkchena',
  `mold_id` varchar(16) DEFAULT NULL COMMENT '??? 1 part vs 1 mold ???',
  `powder_type` varchar(16) DEFAULT NULL COMMENT 'default powder, maybe change when shaping 2018/05/18 lkchena',
  `lot_size_spec` int(11) NOT NULL DEFAULT 0 COMMENT 'total unit count 2018/06/06 lkchena\nin one cart / plate on the stb tool 2020/04/22 lkchena',
  `box_size_spec` int(11) NOT NULL DEFAULT 0 COMMENT 'Trinity maybe use it 2020/04/28 lkchena\ntotal cast count in one cart 2018/06/06 lkchena',
  `wafer_size_spec` int(11) NOT NULL DEFAULT 0 COMMENT 'Trinity maybe use it 2020/04/28 lkchena\n一個 cassette 可以裝多少個 part, 應該定義在 part table, 而不是 carrier 上\n<-- get data from wip table(from part table)',
  `cust_id` varchar(12) DEFAULT NULL,
  `cust_name` varchar(36) DEFAULT NULL,
  `weight_above` decimal(3,1) DEFAULT NULL,
  `weight_below` decimal(3,1) DEFAULT NULL,
  `density_above` decimal(3,1) DEFAULT NULL,
  `density_below` decimal(3,1) DEFAULT NULL,
  `hardness` decimal(3,1) DEFAULT NULL,
  `material` varchar(24) DEFAULT NULL COMMENT '原料',
  `layout_doc` varchar(254) DEFAULT NULL COMMENT 'part layout design document file path',
  `unit_label` int(11) DEFAULT 1 COMMENT 'weight unit: 1: kg 2: ton (tonnage) 2019/12/31 lkchena',
  `unit_type` int(11) DEFAULT 0 COMMENT '2020/05/02 lkchena\n0: by piece(default)\n1: by weigth\nothers.\n\nfn_get_wafer_id will create fullly wafer_id, by weight product will over limitation, so add this field\n',
  `note` varchar(128) DEFAULT NULL,
  `rec_user` varchar(16) DEFAULT NULL,
  `rec_time` varchar(19) DEFAULT NULL,
  `node_id` varchar(16) DEFAULT NULL,
  `node_time` timestamp(6) NULL DEFAULT current_timestamp(6) ON UPDATE current_timestamp(6),
  PRIMARY KEY (`part_id`),
  KEY `idx_mold_id_1_idx` (`mold_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8 AVG_ROW_LENGTH=3276 ROW_FORMAT=DYNAMIC;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `dm_part_bt`
--

LOCK TABLES `dm_part_bt` WRITE;
/*!40000 ALTER TABLE `dm_part_bt` DISABLE KEYS */;
INSERT INTO `dm_part_bt` VALUES ('ACQ51F-1','ACQ51F-1','ROUTE-ZZZ-01',0,'ACQ51F-1','powder1',2100,20,105,'ZZZ-01',NULL,23.0,24.0,8.2,7.9,10.2,NULL,NULL,1,0,NULL,'SYS','2020/04/28 09:37:45',NULL,'2020-04-28 06:49:45.577253'),('IAC69-40040 SILDER','IAC69-40040 SILD','ROUTE-ZZZ-01',0,'SILDER-0235',NULL,16000,0,0,'SANDEN',NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,1,0,NULL,'SYS','2020/04/28 09:37:44',NULL,'2020-04-28 01:37:44.734118'),('SC33-GEAR-01','SC33-GEAR-01','ROUTE-ZZZ-01',0,'ACQ51F-1','S06,41',1200,0,0,'BKM1',NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,1,0,NULL,'SYS','2020/04/28 09:37:44',NULL,'2020-04-28 01:37:44.809394'),('SC34-GEAR-02','SC34-GEAR-02','ROUTE-ZZZ-01',0,'FF-1200','S06,41',1200,0,0,'Custom Real',NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,1,0,NULL,'SYS','2020/04/28 09:37:44',NULL,'2020-04-28 01:37:44.879760'),('SC37-GEAR-03','SC37-GEAR-03','ROUTE-ZZZ-01',0,'FF-1238','S06,40',24000,0,0,'Custom',NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,1,0,NULL,'SYS','2020/04/28 09:37:44',NULL,'2020-04-28 01:37:44.948950'),('WCE42-20025','WCE42-20025','ROUTE-ZZZ-01',0,'ACQ51F-1',NULL,18000,0,0,'HELLWE',NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,1,0,NULL,'SYS','2020/04/28 09:37:44',NULL,'2020-04-28 01:37:45.017691'),('ZZZ-02','ZZZ-02','ROUTE-ZZZ-01',0,'ACQ51F-1','powder1',20000,0,0,'ZZZ-01',NULL,23.0,24.0,8.2,7.9,10.2,NULL,NULL,1,0,NULL,'SYS','2020/04/28 09:37:45',NULL,'2020-04-28 01:37:45.107916');
/*!40000 ALTER TABLE `dm_part_bt` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2020-05-12  8:16:23
