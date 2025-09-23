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
-- Table structure for table `mcs_cart_type_bt`
--

DROP TABLE IF EXISTS `mcs_cart_type_bt`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `mcs_cart_type_bt` (
  `cart_type` varchar(16) NOT NULL COMMENT 'plate_type,cart_type',
  `cnt_total` int(11) NOT NULL DEFAULT 0 COMMENT 'maximal cassette count for one carrier\nL1  -- large   \nM1 -- middle\nS1 -- small   --> 40 ( current )',
  `cnt_box` int(11) NOT NULL DEFAULT 0 COMMENT 'maximal cassette count for one carrier\nL1  -- large   \nM1 -- middle\nS1 -- small   --> 40 ( current )',
  `parent_type` varchar(12) NOT NULL DEFAULT 'SELF' COMMENT 'default: SELF and others',
  `note` varchar(256) DEFAULT NULL,
  `rec_user` varchar(16) DEFAULT NULL,
  `rec_time` datetime DEFAULT NULL,
  `node_id` varchar(16) DEFAULT NULL,
  `node_time` timestamp(6) NULL DEFAULT current_timestamp(6) ON UPDATE current_timestamp(6),
  PRIMARY KEY (`cart_type`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8 ROW_FORMAT=DYNAMIC;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `mcs_cart_type_bt`
--

LOCK TABLES `mcs_cart_type_bt` WRITE;
/*!40000 ALTER TABLE `mcs_cart_type_bt` DISABLE KEYS */;
INSERT INTO `mcs_cart_type_bt` VALUES ('BOX-01',32,0,'PLATE-01','',NULL,NULL,NULL,'2020-02-22 02:43:57.782071'),('BOX-C1',48,0,'CART-01','',NULL,NULL,NULL,'2020-02-22 02:43:52.743043'),('CART-01',420,8,'SELF','',NULL,NULL,NULL,'2020-02-22 02:44:19.051758'),('PLATE-01',1200,12,'SELF','','SYS','2020-02-22 16:41:45',NULL,'2020-02-22 08:41:47.831520'),('PLATE-99',1000,32,'SELF',NULL,'SYS','2020-02-22 16:34:06',NULL,'2020-02-22 08:34:06.567853');
/*!40000 ALTER TABLE `mcs_cart_type_bt` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2020-05-12  8:16:22
