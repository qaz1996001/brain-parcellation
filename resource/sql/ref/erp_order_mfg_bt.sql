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
-- Table structure for table `erp_order_mfg_bt`
--

DROP TABLE IF EXISTS `erp_order_mfg_bt`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `erp_order_mfg_bt` (
  `mfg_no` varchar(16) NOT NULL,
  `order_no` varchar(16) NOT NULL,
  `part_id` varchar(24) NOT NULL,
  `cust_id` varchar(12) DEFAULT NULL COMMENT 'order / part all has cust_id,should we use which one 2018/05/20 lkchena',
  `date_stb` datetime DEFAULT NULL,
  `date_stb_act` datetime DEFAULT NULL COMMENT 'actual stb date 2018/06/08 lkchena',
  `date_due` datetime DEFAULT NULL COMMENT 'due to customer 2018/06/08 lkchena',
  `qty_order` int(11) DEFAULT NULL,
  `qty_stock` int(11) DEFAULT NULL,
  `qty_curr` int(11) DEFAULT NULL,
  `qty_final` int(11) DEFAULT NULL COMMENT 'real product 2018/06/08 lkchena',
  `powder_id` varchar(16) DEFAULT NULL COMMENT '??? wait drop... order table need "powder_id" field ??? 2018/06/08 lkchena',
  `owner1` varchar(16) DEFAULT NULL,
  `note1` varchar(128) DEFAULT NULL,
  `f_create_lot` int(11) DEFAULT 0 COMMENT 'flag:0:not create 1:create. use @erp_stb_create_lot if create lot into ker_wip_rt 2018/06/06 lkchena',
  `f_status` int(11) DEFAULT 0 COMMENT 'default 0:not thing 1: stb complete 2:ongoing  8:complete 9:close(drop/unnormal) 2018/06/08 lkchena',
  `node_id` varchar(16) DEFAULT NULL,
  `node_time` timestamp(6) NULL DEFAULT current_timestamp(6) ON UPDATE current_timestamp(6),
  PRIMARY KEY (`mfg_no`,`order_no`,`part_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 AVG_ROW_LENGTH=5461 ROW_FORMAT=DYNAMIC;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `erp_order_mfg_bt`
--

LOCK TABLES `erp_order_mfg_bt` WRITE;
/*!40000 ALTER TABLE `erp_order_mfg_bt` DISABLE KEYS */;
INSERT INTO `erp_order_mfg_bt` VALUES ('MF181510-04','22-181401-03','SC37-GEAR-03','Custom','2018-10-23 15:02:58',NULL,'2019-01-10 00:00:00',60000,20500,12000,NULL,'S06,40','Mr. Lee',NULL,1,1,NULL,'2020-02-21 05:22:44.217847'),('MF181518-01','22-181501-01','SC33-GEAR-01','BKM1','2018-11-02 15:00:51',NULL,'2018-12-15 00:00:00',88000,12000,9870,NULL,'S06,41','Lucy',NULL,1,1,NULL,'2020-02-21 05:22:44.217847'),('MF181518-03','22-181420-12','SC34-GEAR-02','Custom Real','2018-10-10 15:02:58',NULL,'2018-12-30 00:00:00',12000,0,4500,NULL,'S06,41','Jackal',NULL,1,1,NULL,'2020-02-21 05:22:44.217847');
/*!40000 ALTER TABLE `erp_order_mfg_bt` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2020-05-12  8:16:24
