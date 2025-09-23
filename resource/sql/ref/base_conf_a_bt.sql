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
-- Table structure for table `base_conf_a_bt`
--

DROP TABLE IF EXISTS `base_conf_a_bt`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `base_conf_a_bt` (
  `cate1` varchar(12) NOT NULL DEFAULT '*',
  `cate2` varchar(12) NOT NULL DEFAULT '*',
  `key_id` varchar(12) NOT NULL,
  `key_desc` varchar(32) DEFAULT NULL COMMENT 'description will show at title',
  `value1` varchar(24) DEFAULT NULL,
  `value2` varchar(24) DEFAULT NULL,
  `value3` varchar(24) DEFAULT NULL,
  `note` varchar(128) DEFAULT NULL,
  `rec_user` varchar(16) DEFAULT NULL,
  `rec_time` varchar(19) DEFAULT NULL,
  `node_id` varchar(16) DEFAULT NULL,
  `node_time` timestamp(3) NULL DEFAULT current_timestamp(3) ON UPDATE current_timestamp(3),
  PRIMARY KEY (`cate1`,`cate2`,`key_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COMMENT='different:\nbase_conf_bt: key data\nbase_conf_b_bt: non key data\n2020/01/24 lkchena';
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `base_conf_a_bt`
--

LOCK TABLES `base_conf_a_bt` WRITE;
/*!40000 ALTER TABLE `base_conf_a_bt` DISABLE KEYS */;
INSERT INTO `base_conf_a_bt` VALUES ('AREA_D1','*','C001-A','成型區-A','100',NULL,NULL,NULL,NULL,NULL,NULL,'2020-03-17 02:30:48.830'),('AREA_D1','*','C001-B','成型區-B','110',NULL,NULL,NULL,NULL,NULL,NULL,'2020-03-17 02:30:48.830'),('AREA_D1','*','OIL01','真空含油區','500',NULL,NULL,NULL,NULL,NULL,NULL,'2020-03-17 02:30:48.815'),('AREA_D1','*','OQC-A','品檢區-A','600',NULL,NULL,NULL,NULL,NULL,NULL,'2020-03-17 02:30:48.815'),('AREA_D1','*','OQC-B','品檢區-B','610',NULL,NULL,NULL,NULL,NULL,NULL,'2020-03-17 02:30:48.815'),('AREA_D1','*','PACK-A','包裝區','800',NULL,NULL,NULL,NULL,NULL,NULL,'2020-03-17 02:30:48.815'),('AREA_D1','*','S001','燒結區','200',NULL,NULL,NULL,NULL,NULL,NULL,'2020-03-17 02:30:48.815'),('AREA_D1','*','S003','燒結區-3樓','203',NULL,NULL,NULL,NULL,NULL,NULL,'2020-03-17 02:30:48.815'),('AREA_D1','*','SA01','整形區','300',NULL,NULL,NULL,NULL,NULL,NULL,'2020-03-17 02:30:48.815'),('AREA_D1','*','START-1','開始區','065',NULL,NULL,NULL,NULL,NULL,NULL,'2020-03-17 02:30:48.830'),('AREA_D1','*','VIB01','震盪研磨區','420',NULL,NULL,NULL,NULL,NULL,NULL,'2020-03-17 02:30:48.815'),('SHIFT_D1','*','A','早班','08:00','0',NULL,'旭宏 value1:hh:mm value2:1:cross day 0:not',NULL,NULL,NULL,'2020-02-09 06:26:14.478'),('SHIFT_D1','*','B','中班','16:00','0',NULL,'旭宏 value1:hh:mm value2:1:cross day 0:not',NULL,NULL,NULL,'2020-02-09 06:26:20.131'),('SHIFT_D1','*','C','晚班','00:00','0',NULL,'旭宏 value1:hh:mm value2:1:cross day 0:not','SYS','2020/01/20 09:55:45',NULL,'2020-02-09 06:26:23.926'),('SHIFT_D1_x','*','A','早班','07:00','0',NULL,'value1:hh:mm value2:1:cross day 0:not',NULL,NULL,NULL,'2020-02-09 06:24:24.831'),('SHIFT_D1_x','*','B','晚班','15:00','0',NULL,'value1:hh:mm value2:1:cross day 0:not',NULL,NULL,NULL,'2020-02-09 06:24:26.941'),('SHIFT_D1_x','*','C','大夜班','23:00','1',NULL,'value1:hh:mm value2:1:cross day 0:not','SYS','2020/01/20 09:55:45',NULL,'2020-02-09 06:24:29.526'),('WSTYPE-01','*','10','成形',NULL,NULL,NULL,NULL,NULL,NULL,NULL,'2020-01-24 01:28:01.191'),('WSTYPE-01','*','20','燒結',NULL,NULL,NULL,NULL,NULL,NULL,NULL,'2020-01-24 01:28:01.191'),('WSTYPE-01','*','90','量測',NULL,NULL,NULL,NULL,NULL,NULL,NULL,'2020-01-24 01:28:01.207'),('WSTYPE-01','*','92','量測-重量',NULL,NULL,NULL,NULL,NULL,NULL,NULL,'2020-01-24 01:28:01.207'),('WSTYPE-01','*','93','量測-硬度',NULL,NULL,NULL,NULL,NULL,NULL,NULL,'2020-01-24 01:28:01.207');
/*!40000 ALTER TABLE `base_conf_a_bt` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2020-05-12  8:17:08
